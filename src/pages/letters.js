import React from "react";
import Link from "gatsby-link";
import Helmet from "react-helmet";
import styled from "styled-components";

import CategoryHeader from "../components/CategoryHeader";

const Post = styled.div`
  > h3 {
    margin-bottom: 5px;
  }
  > small {
    font-family: roboto;
    font-size: 14px;
    color: grey;
    font-style: italic;
  }
  > p {
    margin-top: 10px;
  }
`;

const SLink = styled(Link)`
  text-decoration: none;
  color: #28aa55;
  &:hover {
    color: #23984c;
  }
`;

class Letters extends React.Component {
  render() {
    const posts = this.props.data.allMarkdownRemark.edges;
    const categoryTitle = "Letters";
    const categoryDescription = "One day I came upon a realization. There will inevitably be large swaths of my life that are utterly forgotten. Pointers to memories in my brain that have deteriorated to the point where no neural pathways exist. Rich experiences lost in the abyss. Who can recall every moment of their life to vivid detail anyways? I started writing weekly open letters to my close family and friends to serve two purposes; Firstly, a weekly letter to inform the readers on my whereabouts and happenings as I travel around the work, working remotely on data projects. Secondly, a digital repository; storing every little highlight of my life as my own story unfolds.";

    return (
      <div>
        <Helmet title={categoryTitle} />
        <CategoryHeader
          title={categoryTitle}
          description={categoryDescription}
          data= {this.props.data}
        />
        {posts.map(post => {
          if (
            post.node.path !== "/404/"
          ) {
            return (
              <Post key={post.node.frontmatter.path}>
                <h3>
                  <SLink to={post.node.frontmatter.path}>
                    {post.node.frontmatter.title}
                  </SLink>
                </h3>
                <small>{post.node.frontmatter.date}</small>
                <p dangerouslySetInnerHTML={{ __html: post.node.excerpt }} />
              </Post>
            );
          }
        })}
      </div>
    );
  }
}

export default Letters;

export const pageQuery = graphql`
  query LettersQuery {
    allMarkdownRemark(
      sort: { fields: [frontmatter___date], order: DESC },
      filter: {frontmatter: {category: {eq: "Letters"}}}
    ) {
      totalCount
      edges {
        node {
          excerpt(pruneLength: 250)
          frontmatter {
            path
            date(formatString: "MMMM D, YYYY")
            title
            category
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
