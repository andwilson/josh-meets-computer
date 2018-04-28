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
      content: "\u21d2 \0000a0";
      padding-right: 5px;
    }
  }
`;

class Notes extends React.Component {
  render() {
    const posts = this.props.data.allMarkdownRemark.edges;
    const categoryTitle = "Notes";
    const categoryDescription =
      "Below is a bit of an external hard drive for my brain, where I keep notes on the things I've learned while exploring computer science and building projects. I organized my code journals by areas of interest, each note usually has some description of what it is used for and some code snippets I wrote on implementation. Feel free to poke around.";
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
