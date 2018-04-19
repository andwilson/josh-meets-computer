import React from "react";
import Link from "gatsby-link";
import Helmet from "react-helmet";
import styled from "styled-components";

import CategoryHeader from "../components/CategoryHeader";

const SLink = styled(Link)`
  font-family: open sans;
  text-decoration: none;
  color: #28aa55;
  &:hover {
    color: #1d7f3f;
  }
`;

const SectionTitle = styled.h2`
  color: black;
`;

const Ul = styled.ul`
  list-style: none;
  margin-left: 10px;
  padding-left: 0;
  > li {
    padding-left: 1em;
    text-indent: -1em;
    &:before {
      content: '\u21d2 \0000a0';
      padding-right: 5px;
    }
  }
`;

class Notes extends React.Component {
  render() {
    const posts = this.props.data.allMarkdownRemark.edges;
    const categoryTitle = "Notes";
    const categoryDescription =
      "On a seemingly innocuous morning, up in a farmhouse in Vermont, I had an interesting talk about the extensions of the mind. In this paradigm shifting conversation, a very special idea passed on to me. It suggested that everything you put out into the world is an extension of your brain -- whether it's written on a chalk board, typed into a computer, spoken into someone else's ear.. these are your thoughts that once existed in your head but now also exist outside as well. This note section is what I like to consider the external hard drive of my brain, where I keep track of things I've learned so that I can revisit them if approached with a similar problem. ";
    // unique array of sections
    const sections = posts
      .map(post => post.node.frontmatter.section)
      .filter((x, i, a) => a.indexOf(x) == i);

    return (
      <div>
        <Helmet title={categoryTitle} />

        <CategoryHeader
          title={categoryTitle}
          description={categoryDescription}
          data={this.props.data}
        />

        {sections.map(section => {
          return (
            <div>
              <SectionTitle>{section}</SectionTitle>
              {posts.map(post => {
                return (
                  <Ul>
                    {post.node.frontmatter.section == section && (
                      <li>
                        <SLink to={post.node.frontmatter.path}>
                          {post.node.frontmatter.title}
                        </SLink>
                      </li>
                    )}
                  </Ul>
                );
              })}
            </div>
          );
        })}
      </div>
    );
  }
}

export default Notes;

export const pageQuery = graphql`
  query NotesQuery {
    allMarkdownRemark(
      sort: { fields: [frontmatter___date], order: DESC }
      filter: { frontmatter: { category: { eq: "Notes" } } }
    ) {
      totalCount
      edges {
        node {
          excerpt(pruneLength: 300)
          frontmatter {
            path
            date(formatString: "DD MMMM, YYYY")
            title
            category
            section
          }
        }
      }
    }
    avatar: imageSharp(id: { regex: "/avatar.jpg/" }) {
      sizes(maxWidth: 500, grayscale: false) {
        ...GatsbyImageSharpSizes_tracedSVG
      }
    }
  }
`;
